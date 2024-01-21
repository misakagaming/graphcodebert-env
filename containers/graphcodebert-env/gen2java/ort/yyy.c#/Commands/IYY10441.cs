namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: IYY10441_XML_GROUP_PARSE_S       Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:30
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class IYY10441 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10441_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10441_OA WOa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.MYY10441_IA Myy10441Ia;
    GEN.ORT.YYY.MYY10441_OA Myy10441Oa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // REPEATING GROUP VIEW STATUS FIELDS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool ExpGroupList_FL_001;
    int ExpGroupList_PS_001;
    bool ExpGroupList_RF_001;
    public const int ExpGroupList_MM_001 = 10;
    bool ExpGroupList_FL_002;
    int ExpGroupList_PS_002;
    bool ExpGroupList_RF_002;
    public const int ExpGroupList_MM_002 = 10;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020175_esc_flag;
    //       +->   IYY10441_XML_GROUP_PARSE_S        01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp canam_xml (Transient, Optional, Import
    //       !                     only)
    //       !           xml_buffer
    //       !       EXPORTS:
    //       !         Work View exp_error canam_xml (Transient, Export only)
    //       !           xml_return_code
    //       !           xml_message
    //       !           xml_position
    //       !           xml_source
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !         Group View (10) exp_group_list
    //       !           Entity View exp_g_list iyy1_type (Transient, Export
    //       !                       only)
    //       !             tinstance_id
    //       !             treference_id
    //       !             tcreate_user_id
    //       !             tupdate_user_id
    //       !             tkey_attr_text
    //       !             tsearch_attr_text
    //       !             tother_attr_text
    //       !             tother_attr_date
    //       !             tother_attr_time
    //       !             tother_attr_amount
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !   
    //     2 !  NOTE: 
    //     2 !  PURPOSE(CONTINUED)
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     3 !  !!!!!!!!!!!!
    //     3 !  Check for Pre-Post Condition and Return/Reason Code
    //     3 !  
    //     4 !  NOTE: 
    //     4 !  PRE-CONDITION
    //     4 !  Give XML sourse will be parsed as desired attributes 
    //     4 !  POST-CONDITION
    //     4 !  XML will be created
    //     4 !  Return Code = 1, Reason Code = 1
    //     4 !  
    //     5 !  NOTE: 
    //     5 !  PRE-CONDITION
    //     5 !  XML Create Error
    //     5 !  POST-CONDITION
    //     5 !  XML could not created. Check Exp_error canam_xml for details
    //     5 !  Return Code = -70, Reason Code = 1
    //     5 !  
    //     6 !  NOTE: 
    //     6 !  RETURN / REASON CODES
    //     6 !  +1/1 XML Parsed
    //     6 !  -70/1 XML PArse error
    //     6 !  
    //     7 !  NOTE: 
    //     7 !  RELEASE HISTORY
    //     7 !  01_00 01-10-2009 New Release
    //     7 !  
    //     8 !  NOTE: 
    //     8 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     8 !  !!!!!!!!!!!!
    //     8 !  USE <mapper ab>
    //     8 !  
    //     9 !  USE myy10441_xml_group_parse
    //     9 !     WHICH IMPORTS: Work View imp canam_xml TO Work View imp
    //     9 !              canam_xml
    //     9 !     WHICH EXPORTS: Work View exp_error canam_xml FROM Work View
    //     9 !              exp_error canam_xml
    //     9 !                    Work View exp_error iyy1_component FROM Work
    //     9 !              View exp_error iyy1_component
    //     9 !                    Group View  exp_group_list FROM Group View
    //     9 !              exp_group_list
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public IYY10441(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:30";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "IYY10441_XML_GROUP_PARSE_S";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	IYY10441_IA import_view, 
    	IYY10441_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      ++(NestingLevel);
      try {
        f_22020175_init(  );
        f_22020175(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020175(  )
    {
      func_0022020175_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020175" );
      Globdata.GetStateData().SetCurrentABName( "IYY10441_XML_GROUP_PARSE_S" );
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PURPOSE(CONTINUED)                                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    Check for Pre-Post Condition and Return/Reason Code             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    Give XML sourse will be parsed as desired attributes            
      //    POST-CONDITION                                                  
      //    XML will be created                                             
      //    Return Code = 1, Reason Code = 1                                
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    XML Create Error                                                
      //    POST-CONDITION                                                  
      //    XML could not created. Check Exp_error canam_xml for details    
      //    Return Code = -70, Reason Code = 1                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RETURN / REASON CODES                                           
      //    +1/1 XML Parsed                                                 
      //    -70/1 XML PArse error                                           
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 01-10-2009 New Release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <mapper ab>                                                 
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000009" );
      
      Myy10441Ia = (GEN.ORT.YYY.MYY10441_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10441).Assembly,
      	"GEN.ORT.YYY.MYY10441_IA" ));
      Myy10441Oa = (GEN.ORT.YYY.MYY10441_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10441).Assembly,
      	"GEN.ORT.YYY.MYY10441_OA" ));
      Myy10441Ia.ImpCanamXmlXmlBuffer = StringAttr.ValueOf(WIa.ImpCanamXmlXmlBuffer, 4094);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.MYY10441).Assembly,
      	"GEN.ORT.YYY.MYY10441",
      	"Execute",
      	Myy10441Ia,
      	Myy10441Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020175" );
      Globdata.GetStateData().SetCurrentABName( "IYY10441_XML_GROUP_PARSE_S" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000009" );
      WOa.ExpGroupList_MA = IntAttr.ValueOf(Myy10441Oa.ExpGroupList_MA);
      for(Adim1 = 1; Adim1 <= WOa.ExpGroupList_MA; ++(Adim1))
      {
        WOa.ExpGListIyy1TypeTinstanceId[Adim1-1] = TimestampAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTinstanceId[Adim1-1]);
        WOa.ExpGListIyy1TypeTreferenceId[Adim1-1] = TimestampAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTreferenceId[Adim1-1]);
        WOa.ExpGListIyy1TypeTcreateUserId[Adim1-1] = FixedStringAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTcreateUserId[Adim1-1], 8);
        WOa.ExpGListIyy1TypeTupdateUserId[Adim1-1] = FixedStringAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTupdateUserId[Adim1-1], 8);
        WOa.ExpGListIyy1TypeTkeyAttrText[Adim1-1] = FixedStringAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTkeyAttrText[Adim1-1], 4);
        WOa.ExpGListIyy1TypeTsearchAttrText[Adim1-1] = FixedStringAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTsearchAttrText[Adim1-1], 
          20);
        WOa.ExpGListIyy1TypeTotherAttrText[Adim1-1] = FixedStringAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTotherAttrText[Adim1-1], 2)
          ;
        WOa.ExpGListIyy1TypeTotherAttrDate[Adim1-1] = DateAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTotherAttrDate[Adim1-1]);
        WOa.ExpGListIyy1TypeTotherAttrTime[Adim1-1] = TimeAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTotherAttrTime[Adim1-1]);
        WOa.ExpGListIyy1TypeTotherAttrAmount[Adim1-1] = DecimalAttr.ValueOf(Myy10441Oa.ExpGListIyy1TypeTotherAttrAmount[Adim1-1]);
      }
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Myy10441Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Myy10441Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Myy10441Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Myy10441Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Myy10441Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Myy10441Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Myy10441Oa.ExpErrorIyy1ComponentChecksum, 15);
      WOa.ExpErrorCanamXmlXmlReturnCode = FixedStringAttr.ValueOf(Myy10441Oa.ExpErrorCanamXmlXmlReturnCode, 2);
      WOa.ExpErrorCanamXmlXmlMessage = FixedStringAttr.ValueOf(Myy10441Oa.ExpErrorCanamXmlXmlMessage, 80);
      WOa.ExpErrorCanamXmlXmlPosition = DoubleAttr.ValueOf(Myy10441Oa.ExpErrorCanamXmlXmlPosition);
      WOa.ExpErrorCanamXmlXmlSource = FixedStringAttr.ValueOf(Myy10441Oa.ExpErrorCanamXmlXmlSource, 120);
      Myy10441Ia.FreeInstance(  );
      Myy10441Ia = null;
      Myy10441Oa.FreeInstance(  );
      Myy10441Oa = null;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020175_init(  )
    {
      if ( NestingLevel < 2 )
      {
      }
      WOa.ExpErrorCanamXmlXmlReturnCode = "  ";
      WOa.ExpErrorCanamXmlXmlMessage = "                                                                                ";
      WOa.ExpErrorCanamXmlXmlPosition = 0.0;
      WOa.ExpErrorCanamXmlXmlSource = 
        "                                                                                                                        ";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
      WOa.ExpGroupList_MA = 0;
      for(ExpGroupList_PS_001 = 1; ExpGroupList_PS_001 <= 10; ++(ExpGroupList_PS_001))
      {
        WOa.ExpGListIyy1TypeTinstanceId[ExpGroupList_PS_001-1] = "00000000000000000000";
        WOa.ExpGListIyy1TypeTreferenceId[ExpGroupList_PS_001-1] = "00000000000000000000";
        WOa.ExpGListIyy1TypeTcreateUserId[ExpGroupList_PS_001-1] = "        ";
        WOa.ExpGListIyy1TypeTupdateUserId[ExpGroupList_PS_001-1] = "        ";
        WOa.ExpGListIyy1TypeTkeyAttrText[ExpGroupList_PS_001-1] = "    ";
        WOa.ExpGListIyy1TypeTsearchAttrText[ExpGroupList_PS_001-1] = "                    ";
        WOa.ExpGListIyy1TypeTotherAttrText[ExpGroupList_PS_001-1] = "  ";
        WOa.ExpGListIyy1TypeTotherAttrDate[ExpGroupList_PS_001-1] = 00000000;
        WOa.ExpGListIyy1TypeTotherAttrTime[ExpGroupList_PS_001-1] = 00000000;
        WOa.ExpGListIyy1TypeTotherAttrAmount[ExpGroupList_PS_001-1] = DecimalAttr.GetDefaultValue();
      }
      ExpGroupList_PS_001 = 1;
      ExpGroupList_PS_002 = 1;
    }
  }// end class
  
}
