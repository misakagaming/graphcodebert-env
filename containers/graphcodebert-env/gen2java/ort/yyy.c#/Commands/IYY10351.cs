namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: IYY10351_TYPE_LIST_S             Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:36
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
  
  public class IYY10351 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10351_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10351_OA WOa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.MYY10351_IA Myy10351Ia;
    GEN.ORT.YYY.MYY10351_OA Myy10351Oa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // REPEATING GROUP VIEW STATUS FIELDS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool ExpGroupList_FL_001;
    int ExpGroupList_PS_001;
    bool ExpGroupList_RF_001;
    public const int ExpGroupList_MM_001 = 48;
    bool ExpGroupList_FL_002;
    int ExpGroupList_PS_002;
    bool ExpGroupList_RF_002;
    public const int ExpGroupList_MM_002 = 48;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020160_esc_flag;
    //       +->   IYY10351_TYPE_LIST_S              01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_filter iyy1_list (Transient, Mandatory,
    //       !                     Import only)
    //       !           sort_option
    //       !           scroll_type
    //       !           list_direction
    //       !           scroll_amount
    //       !           order_by_field_num
    //       !         Entity View imp_from iyy1_type (Transient, Mandatory,
    //       !                     Import only)
    //       !           tinstance_id
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !         Entity View imp_filter_start iyy1_type (Transient,
    //       !                     Mandatory, Import only)
    //       !           tkey_attr_text
    //       !         Entity View imp_filter_stop iyy1_type (Transient,
    //       !                     Mandatory, Import only)
    //       !           tkey_attr_text
    //       !         Entity View imp_filter iyy1_type (Transient, Mandatory,
    //       !                     Import only)
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !       EXPORTS:
    //       !         Group View (48) exp_group_list
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
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  PURPOSE(CONTINUED)
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     2 !  !!!!!!!!!!!!
    //     2 !  Review Pre-Post Conditions and Return/Reason Codes.
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  PRE-CONDITION
    //     3 !  List type and filters are given.
    //     3 !  POST-CONDITION
    //     3 !  Read data is returned as list.
    //     3 !  Return Code = 1, Reason Code = 1, 11 veya 12
    //     3 !  
    //     4 !  NOTE: 
    //     4 !  RETURN / REASON CODES
    //     4 !  +1/1 List is partially full.
    //     4 !  +1999/1 Other warnings.
    //     4 !  +1/11 List is full, there are records to be listed.
    //     4 !  +1/12 List is empty.
    //     4 !  -1999/1 Other errors.
    //     4 !  
    //     5 !  NOTE: 
    //     5 !  RELEASE HISTORY
    //     5 !  01_00 23-02-1998 New release
    //     5 !  
    //     6 !  NOTE: 
    //     6 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     6 !  !!!!!!!!!!!!
    //     6 !  USE <mapper ab>
    //     6 !  
    //     7 !  USE myy10351_type_list
    //     7 !     WHICH IMPORTS: Work View imp_filter iyy1_list TO Work View
    //     7 !              imp_filter iyy1_list
    //     7 !                    Entity View imp_from iyy1_type TO Entity
    //     7 !              View imp_from iyy1_type
    //     7 !                    Entity View imp_filter_start iyy1_type TO
    //     7 !              Entity View imp_filter_start iyy1_type
    //     7 !                    Entity View imp_filter_stop iyy1_type TO
    //     7 !              Entity View imp_filter_stop iyy1_type
    //     7 !                    Entity View imp_filter iyy1_type TO Entity
    //     7 !              View imp_filter iyy1_type
    //     7 !     WHICH EXPORTS: Group View  exp_group_list FROM Group View
    //     7 !              exp_group_list
    //     7 !                    Work View exp_error iyy1_component FROM Work
    //     7 !              View exp_error iyy1_component
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public IYY10351(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:36";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "IYY10351_TYPE_LIST_S";
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
    	IYY10351_IA import_view, 
    	IYY10351_OA export_view )
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
        f_22020160_init(  );
        f_22020160(  );
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
    public void f_22020160(  )
    {
      func_0022020160_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020160" );
      Globdata.GetStateData().SetCurrentABName( "IYY10351_TYPE_LIST_S" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PURPOSE(CONTINUED)                                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    Review Pre-Post Conditions and Return/Reason Codes.             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    List type and filters are given.                                
      //    POST-CONDITION                                                  
      //    Read data is returned as list.                                  
      //    Return Code = 1, Reason Code = 1, 11 veya 12                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RETURN / REASON CODES                                           
      //    +1/1 List is partially full.                                    
      //    +1999/1 Other warnings.                                         
      //    +1/11 List is full, there are records to be listed.             
      //    +1/12 List is empty.                                            
      //    -1999/1 Other errors.                                           
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <mapper ab>                                                 
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      
      Myy10351Ia = (GEN.ORT.YYY.MYY10351_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10351).Assembly,
      	"GEN.ORT.YYY.MYY10351_IA" ));
      Myy10351Oa = (GEN.ORT.YYY.MYY10351_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10351).Assembly,
      	"GEN.ORT.YYY.MYY10351_OA" ));
      Myy10351Ia.ImpFilterIyy1TypeTsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1TypeTsearchAttrText, 20);
      Myy10351Ia.ImpFilterIyy1TypeTotherAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1TypeTotherAttrText, 2);
      Myy10351Ia.ImpFilterStopIyy1TypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterStopIyy1TypeTkeyAttrText, 4);
      Myy10351Ia.ImpFilterStartIyy1TypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterStartIyy1TypeTkeyAttrText, 4);
      Myy10351Ia.ImpFromIyy1TypeTinstanceId = TimestampAttr.ValueOf(WIa.ImpFromIyy1TypeTinstanceId);
      Myy10351Ia.ImpFromIyy1TypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFromIyy1TypeTkeyAttrText, 4);
      Myy10351Ia.ImpFromIyy1TypeTsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpFromIyy1TypeTsearchAttrText, 20);
      Myy10351Ia.ImpFilterIyy1ListSortOption = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListSortOption, 3);
      Myy10351Ia.ImpFilterIyy1ListScrollType = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListScrollType, 1);
      Myy10351Ia.ImpFilterIyy1ListListDirection = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListListDirection, 1);
      Myy10351Ia.ImpFilterIyy1ListScrollAmount = IntAttr.ValueOf(WIa.ImpFilterIyy1ListScrollAmount);
      Myy10351Ia.ImpFilterIyy1ListOrderByFieldNum = ShortAttr.ValueOf(WIa.ImpFilterIyy1ListOrderByFieldNum);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.MYY10351).Assembly,
      	"GEN.ORT.YYY.MYY10351",
      	"Execute",
      	Myy10351Ia,
      	Myy10351Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020160" );
      Globdata.GetStateData().SetCurrentABName( "IYY10351_TYPE_LIST_S" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Myy10351Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Myy10351Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Myy10351Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Myy10351Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Myy10351Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Myy10351Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Myy10351Oa.ExpErrorIyy1ComponentChecksum, 15);
      WOa.ExpGroupList_MA = IntAttr.ValueOf(Myy10351Oa.ExpGroupList_MA);
      for(Adim1 = 1; Adim1 <= WOa.ExpGroupList_MA; ++(Adim1))
      {
        WOa.ExpGListIyy1TypeTinstanceId[Adim1-1] = TimestampAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTinstanceId[Adim1-1]);
        WOa.ExpGListIyy1TypeTreferenceId[Adim1-1] = TimestampAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTreferenceId[Adim1-1]);
        WOa.ExpGListIyy1TypeTcreateUserId[Adim1-1] = FixedStringAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTcreateUserId[Adim1-1], 8);
        WOa.ExpGListIyy1TypeTupdateUserId[Adim1-1] = FixedStringAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTupdateUserId[Adim1-1], 8);
        WOa.ExpGListIyy1TypeTkeyAttrText[Adim1-1] = FixedStringAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTkeyAttrText[Adim1-1], 4);
        WOa.ExpGListIyy1TypeTsearchAttrText[Adim1-1] = FixedStringAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTsearchAttrText[Adim1-1], 
          20);
        WOa.ExpGListIyy1TypeTotherAttrText[Adim1-1] = FixedStringAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTotherAttrText[Adim1-1], 2)
          ;
        WOa.ExpGListIyy1TypeTotherAttrDate[Adim1-1] = DateAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTotherAttrDate[Adim1-1]);
        WOa.ExpGListIyy1TypeTotherAttrTime[Adim1-1] = TimeAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTotherAttrTime[Adim1-1]);
        WOa.ExpGListIyy1TypeTotherAttrAmount[Adim1-1] = DecimalAttr.ValueOf(Myy10351Oa.ExpGListIyy1TypeTotherAttrAmount[Adim1-1]);
      }
      Myy10351Ia.FreeInstance(  );
      Myy10351Ia = null;
      Myy10351Oa.FreeInstance(  );
      Myy10351Oa = null;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020160_init(  )
    {
      if ( NestingLevel < 2 )
      {
      }
      WOa.ExpGroupList_MA = 0;
      for(ExpGroupList_PS_001 = 1; ExpGroupList_PS_001 <= 48; ++(ExpGroupList_PS_001))
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
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
      ExpGroupList_PS_001 = 1;
      ExpGroupList_PS_002 = 1;
    }
  }// end class
  
}
