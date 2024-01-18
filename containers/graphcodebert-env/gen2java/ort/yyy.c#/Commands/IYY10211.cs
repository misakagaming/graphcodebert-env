namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: IYY10211_CHILD_CREATE_S          Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:57
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
  
  public class IYY10211 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10211_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY10211_OA WOa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.MYY10211_IA Myy10211Ia;
    GEN.ORT.YYY.MYY10211_OA Myy10211Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020108_esc_flag;
    //       +->   IYY10211_CHILD_CREATE_S           01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_reference iyy1_server_data (Transient,
    //       !                     Mandatory, Import only)
    //       !           userid
    //       !           reference_id
    //       !         Entity View imp iyy1_child (Transient, Mandatory,
    //       !                     Import only)
    //       !           cparent_pkey_attr_text
    //       !           ckey_attr_num
    //       !           csearch_attr_text
    //       !           cother_attr_text
    //       !       EXPORTS:
    //       !         Entity View exp iyy1_child (Transient, Export only)
    //       !           cinstance_id
    //       !           creference_id
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
    //     2 !  Review the Pre-Post Conditions and Return/Reason Codes.
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  PRE-CONDITION
    //     3 !  All mandatory fields are given.
    //     3 !  POST-CONDITION
    //     3 !  Record is created.
    //     3 !  Return Code = 1, Reason Code = 1
    //     3 !  
    //     4 !  NOTE: 
    //     4 !  PRE-CONDITION
    //     4 !  The PARENT that the CHILD is related to can' t be found.
    //     4 !  POST-CONDITION
    //     4 !  Record cannot be created.
    //     4 !  Return Code = -40, Reason Code = 4
    //     4 !  
    //     5 !  NOTE: 
    //     5 !  PRE-CONDITION
    //     5 !  Record already exists.
    //     5 !  POST-CONDITION
    //     5 !  Record cannot be created.
    //     5 !  Return Code = -40, Reason Code = 2
    //     5 !  
    //     6 !  NOTE: 
    //     6 !  PRE-CONDITION
    //     6 !  At least one of the record fields is out of the permitted
    //     6 !  value boundaries.
    //     6 !  POST-CONDITION
    //     6 !  Record cannot be created.
    //     6 !  Return Code = -40, Reason Code = 3
    //     6 !  
    //     7 !  NOTE: 
    //     7 !  RETURN / REASON CODES
    //     7 !  +1/1 Record has been created.
    //     7 !  +1999/1 Other warnings.
    //     7 !  -40/4 PARENT could not be found.
    //     7 !  -40/2 CHILD record already exists.
    //     7 !  -40/3 At least one of the CHILD variables is invalid.
    //     7 !  -1999/1 Other errors.
    //     7 !  
    //     8 !  NOTE: 
    //     8 !  RELEASE HISTORY
    //     8 !  01_00 23-02-1998 New release
    //     8 !  
    //     9 !  NOTE: 
    //     9 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     9 !  !!!!!!!!!!!!
    //     9 !  USE <mapper ab>
    //     9 !  
    //    10 !  USE myy10211_child_create
    //    10 !     WHICH IMPORTS: Entity View imp iyy1_child TO Entity View
    //    10 !              imp iyy1_child
    //    10 !                    Work View imp_reference iyy1_server_data TO
    //    10 !              Work View imp_reference iyy1_server_data
    //    10 !     WHICH EXPORTS: Entity View exp iyy1_child FROM Entity View
    //    10 !              exp iyy1_child
    //    10 !                    Work View exp_error iyy1_component FROM Work
    //    10 !              View exp_error iyy1_component
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public IYY10211(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:57";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "IYY10211_CHILD_CREATE_S";
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
    	IYY10211_IA import_view, 
    	IYY10211_OA export_view )
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
        f_22020108_init(  );
        f_22020108(  );
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
    public void f_22020108(  )
    {
      func_0022020108_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020108" );
      Globdata.GetStateData().SetCurrentABName( "IYY10211_CHILD_CREATE_S" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PURPOSE(CONTINUED)                                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    Review the Pre-Post Conditions and Return/Reason Codes.         
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    All mandatory fields are given.                                 
      //    POST-CONDITION                                                  
      //    Record is created.                                              
      //    Return Code = 1, Reason Code = 1                                
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    The PARENT that the CHILD is related to can' t be found.        
      //    POST-CONDITION                                                  
      //    Record cannot be created.                                       
      //    Return Code = -40, Reason Code = 4                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    Record already exists.                                          
      //    POST-CONDITION                                                  
      //    Record cannot be created.                                       
      //    Return Code = -40, Reason Code = 2                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //    At least one of the record fields is out of the permitted       
      //    value boundaries.                                               
      //    POST-CONDITION                                                  
      //    Record cannot be created.                                       
      //    Return Code = -40, Reason Code = 3                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RETURN / REASON CODES                                           
      //    +1/1 Record has been created.                                   
      //    +1999/1 Other warnings.                                         
      //    -40/4 PARENT could not be found.                                
      //    -40/2 CHILD record already exists.                              
      //    -40/3 At least one of the CHILD variables is invalid.           
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
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      
      Myy10211Ia = (GEN.ORT.YYY.MYY10211_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10211).Assembly,
      	"GEN.ORT.YYY.MYY10211_IA" ));
      Myy10211Oa = (GEN.ORT.YYY.MYY10211_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.MYY10211).Assembly,
      	"GEN.ORT.YYY.MYY10211_OA" ));
      Myy10211Ia.ImpReferenceIyy1ServerDataUserid = FixedStringAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataUserid, 8);
      Myy10211Ia.ImpReferenceIyy1ServerDataReferenceId = TimestampAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataReferenceId);
      Myy10211Ia.ImpIyy1ChildCparentPkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ChildCparentPkeyAttrText, 5);
      Myy10211Ia.ImpIyy1ChildCkeyAttrNum = IntAttr.ValueOf(WIa.ImpIyy1ChildCkeyAttrNum);
      Myy10211Ia.ImpIyy1ChildCsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ChildCsearchAttrText, 25);
      Myy10211Ia.ImpIyy1ChildCotherAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ChildCotherAttrText, 25);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.MYY10211).Assembly,
      	"GEN.ORT.YYY.MYY10211",
      	"Execute",
      	Myy10211Ia,
      	Myy10211Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020108" );
      Globdata.GetStateData().SetCurrentABName( "IYY10211_CHILD_CREATE_S" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Myy10211Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Myy10211Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Myy10211Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Myy10211Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Myy10211Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Myy10211Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Myy10211Oa.ExpErrorIyy1ComponentChecksum, 15);
      WOa.ExpIyy1ChildCinstanceId = TimestampAttr.ValueOf(Myy10211Oa.ExpIyy1ChildCinstanceId);
      WOa.ExpIyy1ChildCreferenceId = TimestampAttr.ValueOf(Myy10211Oa.ExpIyy1ChildCreferenceId);
      Myy10211Ia.FreeInstance(  );
      Myy10211Ia = null;
      Myy10211Oa.FreeInstance(  );
      Myy10211Oa = null;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020108_init(  )
    {
      if ( NestingLevel < 2 )
      {
      }
      WOa.ExpIyy1ChildCinstanceId = "00000000000000000000";
      WOa.ExpIyy1ChildCreferenceId = "00000000000000000000";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}

